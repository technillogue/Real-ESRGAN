#!/usr/bin/python3.9
# Copyright (c) 2022 Sylvie Liberman
# pylint: disable=subprocess-run-check
import dataclasses
import json
import logging
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import psycopg

# import redis
import requests
from psycopg.rows import class_row

from realesrgan import RealESRGAN
from config import get_secret

hostname = socket.gethostname()

handler = logging.FileHandler("info.log")
handler.setLevel("INFO")
logging.getLogger().addHandler(handler)
logging.info("starting")
logging.debug("debug")
tee = subprocess.Popen(["tee", "-a", "fulllog.txt"], stdin=subprocess.PIPE)
# Cause tee's stdin to get a copy of our stdin/stdout (as well as that
# of any child processes we spawn)
os.dup2(tee.stdin.fileno(), sys.stdout.fileno())  # type: ignore
os.dup2(tee.stdin.fileno(), sys.stderr.fileno())  # type: ignore

admin_signal_url = "https://imogen-renaissance.fly.dev"

# password, rest = get_secret("REDIS_URL").removeprefix("redis://:").split("@")
# host, port = rest.split(":")
# r = redis.Redis(host=host, port=int(port), password=password)

view_url = "https://mcltajcadcrkywecsigc.supabase.in/storage/v1/object/public/imoges/{slug}.png"

def admin(msg: str) -> None:
    logging.info(msg)
    requests.post(
        f"{admin_signal_url}/admin",
        params={"message": str(msg)},
    )


def stop() -> None:
    paid = "" if os.getenv("FREE") else "paid "
    logging.debug("stopping")
    if os.getenv("POWEROFF"):
        admin(
            f"\N{cross mark}{paid}\N{frame with picture}\N{construction worker}\N{high voltage sign}\N{downwards black arrow} {hostname}"
        )
        subprocess.run(["sudo", "poweroff"])
    elif os.getenv("EXIT"):
        admin(
            f"\N{cross mark}{paid}\N{frame with picture}\N{construction worker}\N{sleeping symbol} {hostname}"
        )
        sys.exit(0)
    else:
        time.sleep(15)


@dataclasses.dataclass
class Prompt:
    prompt_id: int
    prompt: str
    url: str
    slug: str = ""
    params: str = ""
    param_dict: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            self.param_dict = json.loads(self.params or "{}")
            assert isinstance(self.param_dict, dict)
        except (json.JSONDecodeError, AssertionError):
            self.param_dict = {}
        self.safe_prompt = str(int(time.time())) if self.prompt.startswith("http") else self.prompt
        self.slug = f"{self.safe_prompt}_upsampled"


@dataclasses.dataclass
class Result:
    elapsed: int
    filepath: str


def get_prompt(conn: psycopg.Connection) -> Optional[Prompt]:
    conn.execute(
        """UPDATE prompt_queue SET status='pending', assigned_at=null
        WHERE status='assigned' AND assigned_at  < (now() - interval '10 minutes');"""
    )
    selector = os.getenv("SELECTOR")
    selector_cond = "AND selector=%s" if selector else ""
    maybe_id = conn.execute(
        f"SELECT id FROM prompt_queue WHERE status='pending' {selector_cond} ORDER BY signal_ts ASC LIMIT 1;",
        [selector] if selector else [],
    ).fetchone()
    if not maybe_id:
        return None
    prompt_id = maybe_id[0]
    cursor = conn.cursor(row_factory=class_row(Prompt))
    logging.info("getting")
    maybe_prompt = cursor.execute(
        "UPDATE prompt_queue SET status='assigned', assigned_at=now(), hostname=%s WHERE id = %s "
        "RETURNING id AS prompt_id, prompt, params, url;",
        [hostname, prompt_id],
    ).fetchone()
    logging.info("set assigned")
    return maybe_prompt


def main() -> None:
    Path("./input").mkdir(exist_ok=True)
    admin(f"\N{artist palette}\N{construction worker}\N{hiking boot} {hostname}")
    logging.info("starting postgres_jobs on %s", hostname)
    # clear failed instances
    # try to get an id. if we can't, there's no work, and we should stop
    # try to claim it. if we can't, someone else took it, and we should try again
    # generate the prompt
    backoff = 60.0
    generator = None
    # catch some database connection errors
    conn = psycopg.connect(get_secret("DATABASE_URL"), autocommit=True)
    try:
        while 1:
            # try to claim
            prompt = get_prompt(conn)
            if not prompt:
                stop()
                continue
            logging.info("got prompt: %s", prompt)
            try:
                generator, result = handle_item(generator, prompt)
                # success
                start_post = time.time()
                fmt = """UPDATE prompt_queue SET status='uploading', elapsed_gpu=%s, filepath=%s WHERE id=%s;"""
                params = [
                    result.elapsed,
                    result.filepath,
                    prompt.prompt_id,
                ]
                logging.info("set uploading %s", prompt)
                conn.execute(fmt, params)
                post(result, prompt)
                conn.execute(
                    "UPDATE prompt_queue SET status='done' WHERE id=%s",
                    [prompt.prompt_id],
                )
                logging.info("set done, poasting time: %s", time.time() - start_post)
                backoff = 60
            except Exception as e:  # pylint: disable=broad-except
                logging.info("caught exception")
                error_message = traceback.format_exc()
                if prompt:
                    admin(repr(prompt))
                logging.error(error_message)
                admin(error_message)
                if "out of memory" in str(e):
                    sys.exit(137)
                conn.execute(
                    "UPDATE prompt_queue SET errors=errors+1 WHERE id=%s",
                    [prompt.prompt_id],
                )
                time.sleep(backoff)
                backoff *= 1.5
    finally:
        conn.close()


# if there's an initial image, download it from postgres or redis
# pick a slug
# pass maybe raw parameters and initial parameters to the function to get loss and a file
# at this point ideally we need to mark that we generated it, but it wasn't sent yet.
# make a message with the prompt, time, loss, and version
# upload the file, id, and message to imogen based on the url. ideally retry on non-200
Gen = Optional[RealESRGAN]


def handle_item(generator: Gen, prompt: Prompt) -> tuple[Gen, Result]:
    if prompt.prompt.startswith("http"):
        resp = requests.get(prompt.prompt)
    else:
        resp = requests.get(view_url.format(slug=prompt.prompt))
    open(f"inputs/{prompt.safe_prompt}.png", "wb").write(resp.content)
    # if init_image := prompt.param_dict.get("init_image"):
    #     # download the image from redis
    #     open(init_image, "wb").write(r[init_image])
    start_time = time.time()
    if not generator:
        generator = RealESRGAN()
    generator.generate(f"inputs/{prompt.safe_prompt}.png", f"results/{prompt.slug}.png")
    logging.info("generated")
    return generator, Result(
        elapsed=round(time.time() - start_time),
        filepath=f"results/{prompt.slug}.png",
    )


def post(result: Result, prompt: Prompt) -> None:
    minutes, seconds = divmod(result.elapsed, 60)
    bearer = "Bearer " + get_secret("SUPABASE_API_KEY")
    requests.post(
        f"https://mcltajcadcrkywecsigc.supabase.in/storage/v1/object/imoges/{prompt.slug}.png",
        headers={"Authorization": bearer, "Content-Type": "image/png"},
        data=open(result.filepath, mode="rb").read(),
    )
    url = view_url.format(slug=prompt.slug)
    message = f"{url}\nTook {minutes}m{seconds}s to generate"
    admin(message)
    resp = requests.post(
        f"{prompt.url or admin_signal_url}/prompt_message",
        data=message,
        params={"id": str(prompt.prompt_id)},
    )
    logging.info(resp)
    # f = open(result.filepath, mode="rb")
    # for i in range(3):
    #     try:
    #         resp = requests.post(
    #             f"{prompt.url or admin_signal_url}/attachment",
    #             params={"message": message, "id": str(prompt.prompt_id)},
    #             # files={"image": f},
    #         )
    #         logging.info(resp)
    #         break
    #     except requests.RequestException:
    #         logging.info("pausing before retry")
    #         time.sleep(i)

    os.remove(result.filepath)
    # can be retrieved with
    # slug = prompt_queue.filepath.split("/")[1] # bc slug= the directory in filepath
    # requests.get(
    # f"https://mcltajcadcrkywecsigc.supabase.in/storage/v1/object/public/imoges/{prompt.slug}.png"
    # )


if __name__ == "__main__":
    main()

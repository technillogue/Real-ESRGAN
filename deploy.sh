#/bin/bash
set -o xtrace
export v=$1
git commit -a || exit 1
echo "0.0.$v" "$(git log -1 --pretty=%B)"
echo "0.0.$v" "$(git log -1 --pretty=%B)" >> CHANGELOG
docker build -t technillogue/esrgan:0.0.$v .
docker push "technillogue/esrgan:0.0.$v"
sed -i "s#image: technillogue/esrgan:0.0.[[:digit:]]#image: technillogue/esrgan:0.0.$v#g" job.yaml
#kubectl apply -f job.yaml

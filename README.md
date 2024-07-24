# lightspeed docker
## how to run 
clone this repo 
```bash
    git clone https://github.com/thivux/lightspeed_docker.git
```

build & run docker
```bash
docker build -t lightspeed .
docker run -it --name lightspeed-container lightspeed:latest /bin/bash
```

run inference with male or female voice 
```bash
CUDA_VISIBLE_DEVICES=0 python3 infer.py \
--text "Lễ truy điệu Tổng Bí thư Nguyễn Phú Trọng được tổ chức lúc 13h ngày 26/7 \n tại Nhà tang lễ Quốc gia số 5 Trần Thánh Tông, thành phố Hà Nội." \
--gender male \
--output_path "results/male_num.wav" 
```

if text contains multiple paragraphs, they should be separated by a newline character. see example above


## credit 
https://github.com/NTT123/light-speed
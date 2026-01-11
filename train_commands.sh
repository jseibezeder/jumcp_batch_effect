python -u <your_folder>/training/main.py \
--train-file="<your_folder>/source_3_filtered_good_batches.pq" \
--image-path="<your_img_folder>" \
--mapping="<your_folder>/data/class_mapping.json" \
--method="erm" \
--batch-size=16 --batch-size-eval=16 --lr=1e-3 --wd=1e-4 --epochs=200 \
--lr-scheduler="cosine" --warmup=500 --cross-validation=5 \
--workers=4 --model="ResNet50" --dist-url="tcp://127.0.0.1:6031" --tensorboard=True \
--preprocess-img="crop" --normalize=None --image-resolution-train=250 \
--image-resolution-val=250 --patience=15

python -u <your_folder>/training/main.py \
--train-file="<your_folder>/source_3_filtered_good_batches.pq" \
--image-path="<your_img_folder>" \
--mapping="<your_folder>/data/class_mapping.json" \
--method="armcml" \
--batch-size=32 --batch-size-eval=32 --lr=1e-3 --wd=1e-4 --epochs=200 \
--lr-scheduler="cosine" --warmup=500 --cross-validation=5 \
--workers=1 --model="ResNet50" --dist-url="tcp://127.0.0.1:6031" --tensorboard=True \
--preprocess-img="crop" --normalize=None --image-resolution-train=250 \
--image-resolution-val=250 --adapt-bn=False --cml-hidden-dim=64 --grad-acc=2 --patience=15

python -u <your_folder>/training/main.py \
--train-file="<your_folder>/source_3_filtered_good_batches.pq" \
--image-path="<your_img_folder>" \
--mapping="<your_folder>/data/class_mapping.json" \
--method="armbn" \
--batch-size=64 --batch-size-eval=64 --lr=1e-3 --wd=1e-4 --epochs=200 \
--lr-scheduler="cosine" --warmup=500 --cross-validation=5 \
--workers=1 --model="ResNet50" --dist-url="tcp://127.0.0.1:6031" --tensorboard=True \
--preprocess-img="crop" --normalize=None --image-resolution-train=250 \
--image-resolution-val=250 --patience=15

python -u <your_folder>/training/main.py \
--train-file="<your_folder>/source_3_filtered_good_batches.pq" \
--image-path="<your_img_folder>" \
--mapping="<your_folder>/data/class_mapping.json" \
--method="armll" \
--batch-size=16 --batch-size-eval=16 --lr=1e-3 --wd=1e-4 --epochs=200 \
--lr-scheduler="cosine" --warmup=500 --cross-validation=5 \
--workers=1 --model="ResNet50" --dist-url="tcp://127.0.0.1:6031" --tensorboard=True \
--preprocess-img="crop" --normalize=None --image-resolution-train=250 \
--image-resolution-val=250 --grad-acc=4 --patience=15

python -u <your_folder>/training/main.py \
--train-file="<your_folder>/source_3_filtered_good_batches.pq" \
--image-path="<your_img_folder>" \
--mapping="<your_folder>/data/class_mapping.json" \
--method="erm" \
--batch-size=16 --batch-size-eval=16 --lr=1e-3 --wd=1e-4 --epochs=200 \
--lr-scheduler="cosine" --warmup=500 --cross-validation=5 \
--workers=4 --model="ResNet50" --dist-url="tcp://127.0.0.1:6031" --tensorboard=True \
--preprocess-img="rotate" --normalize=None --image-resolution-train=250 \
--image-resolution-val=250 --patience=15

python -u <your_folder>/training/main.py \
--train-file="<your_folder>/source_3_filtered_good_batches.pq" \
--image-path="<your_img_folder>" \
--mapping="<your_folder>/data/class_mapping.json" \
--method="armcml" \
--batch-size=32 --batch-size-eval=32 --lr=1e-3 --wd=1e-4 --epochs=200 \
--lr-scheduler="cosine" --warmup=500 --cross-validation=5 \
--workers=1 --model="ResNet50" --dist-url="tcp://127.0.0.1:6031" --tensorboard=True \
--preprocess-img="rotate" --normalize=None --image-resolution-train=250 \
--image-resolution-val=250 --adapt-bn=False --cml-hidden-dim=64 --grad-acc=2 --patience=15

python -u <your_folder>/training/main.py \
--train-file="<your_folder>/source_3_filtered_good_batches.pq" \
--image-path="<your_img_folder>" \
--mapping="<your_folder>/data/class_mapping.json" \
--method="armbn" \
--batch-size=64 --batch-size-eval=64 --lr=1e-3 --wd=1e-4 --epochs=200 \
--lr-scheduler="cosine" --warmup=500 --cross-validation=5 \
--workers=1 --model="ResNet50" --dist-url="tcp://127.0.0.1:6031" --tensorboard=True \
--preprocess-img="rotate" --normalize=None --image-resolution-train=250 \
--image-resolution-val=250 --patience=15

python -u <your_folder>/training/main.py \
--train-file="<your_folder>/source_3_filtered_good_batches.pq" \
--image-path="<your_img_folder>" \
--mapping="<your_folder>/data/class_mapping.json" \
--method="armll" \
--batch-size=16 --batch-size-eval=16 --lr=1e-3 --wd=1e-4 --epochs=200 \
--lr-scheduler="cosine" --warmup=500 --cross-validation=5 \
--workers=1 --model="ResNet50" --dist-url="tcp://127.0.0.1:6031" --tensorboard=True \
--preprocess-img="rotate" --normalize=None --image-resolution-train=250 \
--image-resolution-val=250 --grad-acc=4 --patience=15

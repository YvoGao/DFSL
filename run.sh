for DATASET in OxfordPets EuroSAT UCF101 SUN397 Caltech101 DescribableTextures FGVCAircraft Food101 OxfordFlowers StanfordCars ImageNet
# for DATASET in ImageNet
do
    for shot in 16  4  1
    do CUDA_VISIBLE_DEVICES=6 python train.py --dataset $DATASET \
        --feature_extractor clip \
        --num_shots $shot

    done
done


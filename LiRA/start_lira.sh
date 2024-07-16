
train_size=25000
folder_tag=lira
res_folder=lira-$folder_tag-fullMember-$train_size
gpu=0
declare -i total_models=128
declare -i index=0
index=$total_models-1
for tag in $(seq 0 $index)
do
    python train_shadow.py --save_tag $tag --total_models $total_models --train_size $train_size \
            --folder_tag $folder_tag  --res_folder $res_folder

     python lira-inference.py --org_path shadow-$folder_tag-trainSize-$train_size-fullMember/model_best-$tag.pth.tar \
    --save_tag $tag --train_size $train_size  --res_folder $res_folder
done

python lira-score.py --res_folder $res_folder
python lira-inference-defense.py --org_path ./final-all-models/lira-trainSize-$train_size.pth.tar \
     --train_size $train_size  --res_folder lira-$folder_tag-defense-fullMember-$train_size
python lira-score.py --res_folder lira-$folder_tag-defense-fullMember-$train_size
python lira-plot.py --shadow_data_path $res_folder --test_data_path lira-$folder_tag-defense-fullMember-$train_size
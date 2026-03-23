# todo tpvformer eval lidarseg
python /vepfs-mlp2/c20250502/haoce/wangyushen/TPVFormer/eval.py \
    --py-config /vepfs-mlp2/c20250502/haoce/wangyushen/TPVFormer/config/tpv_lidarseg_custom.py \
    --work-dir /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/tpvformer/outpus \
    --ckpt-path /c20250502/wangyushen/Weights/tpvformer/tpv10_lidarseg_v2.pth \

# todo eval occpancy
python /vepfs-mlp2/c20250502/haoce/wangyushen/TPVFormer/eval.py \
    --py-config /vepfs-mlp2/c20250502/haoce/wangyushen/TPVFormer/config/tpv04_occupancy_custom.py \
    --work-dir /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/tpvformer/outpus \ 
    --ckpt-path /c20250502/wangyushen/Weights/tpvformer/tpv04_occupancy_v2.pth
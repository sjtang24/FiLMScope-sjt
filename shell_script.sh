# Obtain the reconstruction for experiments on the eye video
(python reconstruction_scripts/video-recon-experiments/reconstruct.py \
    --only_gold_standards \
    --gpu 0 \
    --sample_name "eye_video" \
    --save_final --filter "median" &&
python reconstruction_scripts/video-recon-experiments/reconstruct.py \
    --iters 1 \
    --gpu 0 \
    --sample_name "eye_video" \
    --save_final \
    --filter "median" &&
python reconstruction_scripts/video-recon-experiments/reconstruct.py \
    --iters 5 \
    --gpu 0 \
    --sample_name "eye_video" \
    --save_final \
    --filter "median" &&
python reconstruction_scripts/video-recon-experiments/reconstruct.py \
    --iters 10 \
    --gpu 0 \
    --sample_name "eye_video" \
    --save_final \
    --filter "median") > logs/eye.log 2>&1 &


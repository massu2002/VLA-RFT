NUM_NODES=3 \
NODE_INDEX=0 \
SWEEP_PRESET=core \
RUN_NAME=phase1_residual_spatial \
RFT_RUN_NAME=phase1_residual_spatial_rft \
RFT_CKPT_ROOT=checkpoints/libero/PixelResidualWM-RFT \
RFT_STEPS=400 \
bash scripts/libero/phase1/run_residual_wm_rft_only.sh spatial

NUM_NODES=3 \
NODE_INDEX=1 \
SWEEP_PRESET=core \
RUN_NAME=phase1_residual_spatial \
RFT_RUN_NAME=phase1_residual_spatial_rft \
RFT_CKPT_ROOT=checkpoints/libero/PixelResidualWM-RFT \
RFT_STEPS=400 \
bash scripts/libero/phase1/run_residual_wm_rft_only.sh spatial

NUM_NODES=3 \
NODE_INDEX=2 \
SWEEP_PRESET=core \
RUN_NAME=phase1_residual_spatial \
RFT_RUN_NAME=phase1_residual_spatial_rft \
RFT_CKPT_ROOT=checkpoints/libero/PixelResidualWM-RFT \
RFT_STEPS=400 \
bash scripts/libero/phase1/run_residual_wm_rft_only.sh spatial
# clone official repositories
git clone git@github.com:lingyanruan/DRBNet.git executor/defocus_deblurring/tools/DRBNet
git clone git@github.com:codeslake/IFAN.git executor/defocus_deblurring/tools/IFAN
git clone git@github.com:swz30/Restormer.git executor/defocus_deblurring/tools/Restormer

git clone git@github.com:IDKiro/DehazeFormer.git executor/dehazing/tools/DehazeFormer
git clone git@github.com:google-research/maxim.git executor/dehazing/tools/maxim
git clone git@github.com:RQ-Wu/RIDCP_dehazing.git executor/dehazing/tools/RIDCP_dehazing
git clone git@github.com:Andrew0613/X-Restormer.git executor/dehazing/tools/X-Restormer

git clone git@github.com:swz30/MPRNet.git  executor/denoising/tools/MPRNet
git clone git@github.com:JingyunLiang/SwinIR.git executor/denoising/tools/SwinIR

git clone git@github.com:jiaxi-jiang/FBCNN.git executor/jpeg_compression_artifact_removal/tools/FBCNN

git clone git@github.com:XPixelGroup/HAT.git executor/super_resolution/tools/HAT
git clone git@github.com:XPixelGroup/DiffBIR.git executor/super_resolution/tools/DiffBIR

cd executor/super_resolution/tools/DiffBIR
git checkout 7bd5675
cd ../../../..

# prepare custom scripts to adapt to our framework
mv installation/custom_tool_scripts/drbnet_run.py executor/defocus_deblurring/tools/DRBNet/run.py
mv installation/custom_tool_scripts/dehazeformer_inference.py executor/dehazing/tools/DehazeFormer/inference.py
mv installation/custom_tool_scripts/xrestormer_inference.py executor/dehazing/tools/X-Restormer/xrestormer/inference.py
mv installation/custom_tool_scripts/swinir_inference.py executor/denoising/tools/SwinIR/inference.py
mv installation/custom_tool_scripts/fbcnn_inference.py executor/jpeg_compression_artifact_removal/tools/FBCNN/inference.py
mv installation/custom_tool_scripts/hat_inference.py executor/super_resolution/tools/HAT/hat/inference.py

# some repos are used for multiple single-degradation restorations tasks; use symlinks to save space
ln -s $(pwd)/executor/dehazing/tools/maxim executor/denoising/tools/
ln -s $(pwd)/executor/defocus_deblurring/tools/Restormer executor/denoising/tools/
ln -s $(pwd)/executor/dehazing/tools/X-Restormer executor/denoising/tools/

ln -s $(pwd)/executor/dehazing/tools/maxim executor/deraining/tools/
ln -s $(pwd)/executor/denoising/tools/MPRNet executor/deraining/tools/
ln -s $(pwd)/executor/defocus_deblurring/tools/Restormer executor/deraining/tools/
ln -s $(pwd)/executor/dehazing/tools/X-Restormer executor/deraining/tools/

ln -s $(pwd)/executor/denoising/tools/SwinIR executor/jpeg_compression_artifact_removal/tools/

ln -s $(pwd)/executor/dehazing/tools/maxim executor/motion_deblurring/tools/
ln -s $(pwd)/executor/denoising/tools/MPRNet executor/motion_deblurring/tools/
ln -s $(pwd)/executor/defocus_deblurring/tools/Restormer executor/motion_deblurring/tools/
ln -s $(pwd)/executor/dehazing/tools/X-Restormer executor/motion_deblurring/tools/

ln -s $(pwd)/executor/denoising/tools/SwinIR executor/super_resolution/tools/
ln -s $(pwd)/executor/dehazing/tools/X-Restormer executor/super_resolution/tools/

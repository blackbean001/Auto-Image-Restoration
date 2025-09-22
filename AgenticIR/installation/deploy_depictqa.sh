git clone https://github.com/XPixelGroup/DepictQA.git

mv installation/custom_depictqa_scripts/app_eval.py DepictQA/src/
mv installation/custom_depictqa_scripts/app_comp.py DepictQA/src/

mkdir DepictQA/experiments/agenticir
mv installation/custom_depictqa_scripts/config_eval.yaml DepictQA/experiments/agenticir/
mv installation/custom_depictqa_scripts/config_comp.yaml DepictQA/experiments/agenticir/

mkdir DepictQA/weights/delta -p

mv installation/tune_depictqa/* DepictQA/experiments/agenticir
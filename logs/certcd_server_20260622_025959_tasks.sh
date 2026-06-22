real assist_09 ncdm logs/certcd_assist_09_ncdm.log /home/zsh/anaconda3/envs/xph_env/bin/python -m certcd.run --dataset assist_09 --model ncdm --output-dir outputs/certcd_run
real assist_09 kancd logs/certcd_assist_09_kancd.log /home/zsh/anaconda3/envs/xph_env/bin/python -m certcd.run --dataset assist_09 --model kancd --output-dir outputs/certcd_run
real assist_17 ncdm logs/certcd_assist_17_ncdm.log /home/zsh/anaconda3/envs/xph_env/bin/python -m certcd.run --dataset assist_17 --model ncdm --output-dir outputs/certcd_run
real assist_17 kancd logs/certcd_assist_17_kancd.log /home/zsh/anaconda3/envs/xph_env/bin/python -m certcd.run --dataset assist_17 --model kancd --output-dir outputs/certcd_run
real junyi ncdm logs/certcd_junyi_ncdm.log /home/zsh/anaconda3/envs/xph_env/bin/python -m certcd.run --dataset junyi --model ncdm --output-dir outputs/certcd_run
real junyi kancd logs/certcd_junyi_kancd.log /home/zsh/anaconda3/envs/xph_env/bin/python -m certcd.run --dataset junyi --model kancd --output-dir outputs/certcd_run
real nips34_retricd_small ncdm logs/certcd_nips34_retricd_small_ncdm.log /home/zsh/anaconda3/envs/xph_env/bin/python -m certcd.run --dataset nips34_retricd_small --model ncdm --output-dir outputs/certcd_run
real nips34_retricd_small kancd logs/certcd_nips34_retricd_small_kancd.log /home/zsh/anaconda3/envs/xph_env/bin/python -m certcd.run --dataset nips34_retricd_small --model kancd --output-dir outputs/certcd_run
synth synth ncdm logs/certcd_synth_ncdm.log /home/zsh/anaconda3/envs/xph_env/bin/python -m certcd.run_synthetic --output-dir outputs/certcd_run/synth --model ncdm
synth synth kancd logs/certcd_synth_kancd.log /home/zsh/anaconda3/envs/xph_env/bin/python -m certcd.run_synthetic --output-dir outputs/certcd_run/synth --model kancd
cat assist_09 ncdm logs/certcd_cat_assist_09.log /home/zsh/anaconda3/envs/xph_env/bin/python -m certcd.cat --dataset assist_09 --model ncdm --output-dir outputs/certcd_run/cat
cat assist_17 ncdm logs/certcd_cat_assist_17.log /home/zsh/anaconda3/envs/xph_env/bin/python -m certcd.cat --dataset assist_17 --model ncdm --output-dir outputs/certcd_run/cat
cat nips34_retricd_small ncdm logs/certcd_cat_nips34_retricd_small.log /home/zsh/anaconda3/envs/xph_env/bin/python -m certcd.cat --dataset nips34_retricd_small --model ncdm --output-dir outputs/certcd_run/cat

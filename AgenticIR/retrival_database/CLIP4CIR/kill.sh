ps aux|grep clip_fine_tune|awk '{print $2}'|xargs kill -9
ps aux|grep combiner_train.py | awk '{print $2}'|xargs kill -9

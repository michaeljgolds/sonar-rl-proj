# -*- coding: utf-8 -*-

import envV2 as ENV
import numpy as np
import gc
from pympler import muppy, summary
import objgraph

from keras.utils import plot_model

crashes_arr = np.zeros(5)


env = ENV.sonarEnv(speed=0.5,dronesize=0.1)
crashes = 0
for i in range(100):
    a = env.step(0)
    env.render_for_nips(i)
    if a[2]:
        crashes = crashes+1
        env.reset()
    print(i)
    
    
    # all_objects = muppy.get_objects()
    # sum1 = summary.summarize(all_objects)
    # summary.print_(sum1)
    
    # plot_model(env.gan.generator,to_file=str(i)+'__gen.png',show_shapes=True)
    # plot_model(env.gan.generator_model,to_file=str(i)+'__gen_model.png',show_shapes=True)
            
# crashes_arr[int(ds*5)-1] = crashes
env.close()

print(crashes_arr)
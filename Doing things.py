# Simple script to demonstrate how to use the environment as a black box.

# Import the environment
import envV2 as ENV

#Load the environment, it has a number of variables that can be initailized.
#Here we just set the movement speed of the drone and drone size radius.
env = ENV.sonarEnv(speed=0.5,dronesize=0.1)


# This loop just moves forward for 100 steps, if the drone crashes we reset.
for i in range(100):
    
    # Step function has the action as input: 0 - forward, 1 - backwards, 2 - turn left, 3 - turn right
    # It returns a vector of the state, the reward, the `finished' boolean
    a = env.step(0)
    
    # Render will plot the state as a curve, and also plots a top down plot of the trees
    env.render(i)
    
    # If `finished' we reset
    if a[2]:
        env.reset()
    

# Frees some memory when finished with the environment
env.close()
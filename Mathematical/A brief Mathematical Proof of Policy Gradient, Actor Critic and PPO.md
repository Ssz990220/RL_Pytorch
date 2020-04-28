A brief Mathematical Proof of Policy Gradient, Actor Critic and PPO

\1.   Policy Gradient

The main idea of Policy Gradient is to maximize the expectation of reward gaining from a trajectory sampled from a policy. To maximize the total reward, we use gradient decent algorithm.

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png)

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image004.png)

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image006.png) 

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image008.png) 

Thus,

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image010.png)

Take a further step:

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image012.png)

If we sample trajectory by let the “actor” with parameter ![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image014.png) to interact with the environment, let’s say, N trajectory is generated in form ![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image016.png) , the probability of each trajectory is

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image018.png)

Plug into eqn(2)

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image020.png)

Then we expand the ![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image022.png) item

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image024.png)

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image026.png)

We can get

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image028.png)

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image030.png)

With eqn(3):

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image032.png)

Hence, the loss function of Policy Gradient is

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image034.png)

\2.   Policy Gradient with Baseline

For some specific environment, the reward is always true, which is unbeneficial for the network to converge.

Rewriting eqn(4) by moving reward into gradient and adding a bias to reward

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image036.png)

Here, b is a constant or function independent of ![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image014.png), which indicates ![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image038.png).

The equation remains valid.

 

Replace total reward ![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image040.png) with discounted total gain ![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image042.png), and replace b with a function ![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image044.png), which means discounted reward before ![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image046.png), ![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image048.png)

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image050.png) ![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image052.png).

Other forms of replacement are, replace ![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image054.png) with advantages ![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image056.png).

 

The key difference between baselined policy gradient with traditional policy gradient algorithm mentioned in section 1 is, it can be updated after on each step instead of over the whole trajectory. 

*The core of those to algorithms is identical. The only difference is, algorithm in section 1 collects gradient of each step over the whole trajectory and execute gradient decent afterwards, while the algorithm in section 2 updated at each step.*

 

\3.    Actor Critic

Following the baseline policy gradient, we replace the weight ![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image054.png) with *TD(0)* error

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image058.png)

![img](file:///C:/Users/91354/AppData/Local/Temp/msohtmlclip1/01/clip_image060.png), it is also a neural network which called *“Critic”* .

We update the value network at each step by minimizing TD-error.

And update Actor network with policy gradient.

Then we get Actor Critic algorithm

\4.    Importance Sampling

 
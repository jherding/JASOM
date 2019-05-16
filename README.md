# JASOM - just another self-organizing map
JASOM implements the idea of [Kohonen's self-organizing maps (SOM)](https://en.wikipedia.org/wiki/Self-organizing_map) with the help of TensorFlow. There are already plenty of good implementations of SOM - also using TensorFlow - however, none of them allows for user-specified distance metrics and only the [MDP toolkit](http://mdp-toolkit.sourceforge.net/) implements [growing neural gas (GNG)](https://en.wikipedia.org/wiki/Neural_gas) - a special form of SOM.

So, my motivation for this project was to provide an implementation of SOM, and in particular also of GNG, that allows the user to specify a custom distance metric... and I wanted to learn low-level TensorFlow :)

Work in progress...
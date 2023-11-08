# chrono-TunerINN
Simulator for Tuner INN project using chrono

## Installation of chrono

If your chrono environment does not have some packages, please try the following step.

Try creating an environment from this yaml file in this repo using:
'''bash
conda env create -f environment.yml -n chrono
'''

Download this version of pychrono: https://anaconda.org/projectchrono/pychrono/8.0.0/download/linux-64/pychrono-8.0.0-py39_1.tar.bz2

Once it successfully creates the environment, activate it and do:
'''bash
conda install pychrono-8.0.0-py39_1.tar.bz2
'''

## Code description
* main.py: Run the simulation on the maps from maps directory. Create the control input and feed it to env.step()
* chrono_env/environment.py: Open-AI Gym-like environment. 'make' inits vehicle, terrain, and visualization. 'step' updates the state of the vehicle (env.my_hmmwv.state) and control input. It then  obtains and prints the steering angles of 4 wheels.
* chrono_env/utils.py: This is the file that implements 'get_toe_in', which obtains the steering angle of a wheel.





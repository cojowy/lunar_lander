ENV SETUP:
Todo: requirements.txt
$source activate <env_name>

CREATE DATA:
This command runs data_gen script for 10m.
$timeout -sHUP 10m python ./lunar_lander_data_gen.py

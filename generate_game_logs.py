import subprocess, os

f = open('last_game_num.txt', 'r')

# the number of the last game generated
last_game_num = int(f.read())

num_to_generate = 1000

for i in range(num_to_generate):
	game_num = last_game_num + i + 1

	bashCommand = 'advanced_ai/bin/2048 > game_logs/game' + str(game_num) + '.log'

	print('running:',bashCommand)
	os.system(bashCommand)

	bashCommand = 'echo ' + str(game_num) + ' > last_game_num.txt'
	os.system(bashCommand)

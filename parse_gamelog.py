import numpy as np

def parse_board(x):
	x = x.split('\n')[1:5]
	try:
		x[3] = x[3].split('\n')[0]
	except Exception as e:
		import pdb ; pdb.set_trace()
		raise e
	
	for i in range(4):
		x[i] = [int(y) for y in x[i].split()]
	return np.array(x)

def parse_move(x):
	return int(x.split('Take move: ')[1].split('\n')[0])

def main():
	f = open('game.log', 'r')

	text = f.read()

	games = text.split('#')
	games = games[1:]

	parsed = []

	for chunk in games:
		board = parse_board(chunk)
		move = parse_move(chunk)
		parsed.append((board, move))

	print parsed

if __name__ == '__main__':
	main()
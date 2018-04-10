from learning_agent import QLearningAgent

def main():
	# agent = QLearningAgent(model_dest='/tmp/2048_model0.ckpt')
	agent = QLearningAgent(model_dest='/tmp/2048_model1.ckpt')
	# agent = QLearningAgent(model_dest='/tmp/2048_model2.ckpt')
	agent.play()

if __name__ == '__main__':
	main()

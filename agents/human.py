import sys


# getch method
try:
    from msvcrt import getch
except ImportError:
    def getch():
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


class Human:
    def __init__(self, env):
        self.env = env

    def train(self, episodes=0, render=False):
        pass

    def test(self, render=True):
        while True:
            self.env.reset()
            self.env.render(wait=0)
            done = False
            while not done:
                action = None
                while action not in self.env.action_space:
                    key = ord(getch())
                    if key == 3:
                        sys.exit(0)
                    if key == 27 and ord(getch()) == 91:
                        key = ord(getch())
                        action = key - 65
                state, reward, done = self.env.step(action)
                print('state: {}, reward: {:.2f}'.format(state, reward))
                self.env.render(wait=0)
            print('Terminal')

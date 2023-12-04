import sys
import threading
import time


class SimpleSpinner:
    def __init__(self, message='Processing', style='moon'):
        self.text = message
        self.is_running = False
        self.style = style

        # Different styles of spinners
        self.spinners = {
            'default': ['|', '/', '-', '\\', '-', '\\'],
            'dots': ['.', '..', '...', '....'],
            'block': ['▁', '▃', '▄', '▆', '▇', '█', '▇', '▆', '▄', '▃'],
            'circle': ['◐', '◓', '◑', '◒'],
            'arrow': ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'],
            'moon': ['🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘'],
            'clock': ['🕛', '🕐', '🕑', '🕒', '🕓', '🕔', '🕕', '🕖', '🕗', '🕘', '🕙', '🕚'],
            'bouncing_bar': ['▶▶▶▶▶◀◀', '▶▶▶▶◀◀◀', '▶▶▶◀◀◀◀', '▶▶◀◀◀◀◀', '▶◀◀◀◀◀◀', '◀◀◀◀◀◀▶', '◀◀◀◀◀▶▶', '◀◀◀◀▶▶▶',
                             '◀◀◀▶▶▶▶', '◀◀▶▶▶▶▶', '◀▶▶▶▶▶▶'],
            'heart': ['💗', '💓', '💖', '💘', '💝'],
            'wave': ['~    ', ' ~   ', '  ~  ', '   ~ ', '    ~', '   ~ ', '  ~  ', ' ~   ']
        }

    def start(self):
        self.is_running = True
        threading.Thread(target=self._spin).start()

    def stop(self):
        self.is_running = False
        time.sleep(0.1)  # Allow last spin iteration to finish
        sys.stdout.write('\r' + ' ' * (len(self.text) + 2) + '\r')  # Clear line
        sys.stdout.flush()

    def update(self, message, style=None):
        restart = False
        if self.is_running:
            restart = True
            self.stop()

        self.text = message
        if style:
            self.style = style

        if restart:
            self.start()

    def _spin(self):
        spinner = self.spinners.get(self.style, self.spinners['default'])
        idx = 0
        while self.is_running:
            sys.stdout.write(f'\r{self.text} {spinner[idx]}')
            sys.stdout.flush()
            idx = (idx + 1) % len(spinner)
            time.sleep(0.1)
        sys.stdout.write('\r')  # Clear the spinner line

# Usage:
# spinner = SimpleSpinner(style='block')
# spinner.start()
# time.sleep(5)
# spinner.stop()

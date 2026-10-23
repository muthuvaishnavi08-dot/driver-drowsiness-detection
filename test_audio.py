import platform
import time
import sys

print("Testing audio system...")
print(f"Platform: {platform.system()}")

if platform.system() == "Windows":
    import winsound
    print("Playing Windows beep...")
    winsound.Beep(1000, 500)  # Frequency, Duration
    print("Done!")
else:
    try:
        import pygame
        pygame.mixer.init()
        print("Playing pygame beep...")
        
        import numpy as np
        import scipy.io.wavfile as wav
        
        # Generate beep
        sample_rate = 44100
        duration = 0.5
        freq = 1000
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        wave = np.sin(2 * np.pi * freq * t) * 0.5
        sound = (wave * 32767).astype(np.int16)
        
        # Play
        sound_arr = np.zeros((len(sound), 2), dtype=np.int16)
        sound_arr[:, 0] = sound
        sound_arr[:, 1] = sound
        
        pygame_sound = pygame.sndarray.make_sound(sound_arr)
        pygame_sound.play()
        
        time.sleep(1)
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
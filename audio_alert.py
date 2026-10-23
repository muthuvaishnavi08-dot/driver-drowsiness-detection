import platform
import time
import winsound
import threading

class AudioAlert:
    def __init__(self):
        self.is_playing = False
        print("✓ Audio Alert System Initialized")
        
    def play_alert(self):
        """Play alert beep - simple and direct"""
        if self.is_playing:
            return True
            
        self.is_playing = True
        
        def beep_loop():
            # Double beep pattern
            winsound.Beep(1000, 300)  # Frequency 1000Hz, Duration 300ms
            
            # Small delay between beeps
            if self.is_playing:
                time.sleep(0.15)
                winsound.Beep(1500, 300)  # Higher pitch second beep
                
            # Wait before repeating
            for _ in range(8):  # Wait ~0.4 seconds
                if not self.is_playing:
                    break
                time.sleep(0.05)
                
            self.is_playing = False
        
        # Start beep in background
        thread = threading.Thread(target=beep_loop, daemon=True)
        thread.start()
        return True
        
    def stop_alert(self):
        """Stop the alert"""
        self.is_playing = False

# Test function
def test_sound():
    """Test if sound works"""
    print("\n" + "="*50)
    print("TESTING AUDIO ALERT")
    print("="*50)
    
    print("\nPlaying single beep...")
    winsound.Beep(800, 500)
    print("Did you hear the beep?")
    
    print("\nPlaying double beep...")
    winsound.Beep(1000, 250)
    time.sleep(0.2)
    winsound.Beep(1200, 250)
    print("Did you hear 2 beeps?")
    
    print("\nPlaying alarm pattern...")
    for i in range(3):
        winsound.Beep(800, 200)
        time.sleep(0.15)
        winsound.Beep(1200, 200)
        time.sleep(0.3)
        
    print("\n✓ Sound test complete!")
    return True

if __name__ == "__main__":
    test_sound()
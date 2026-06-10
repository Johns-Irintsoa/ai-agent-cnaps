import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class TypewriterService {
  play(fullText: string, onUpdate: (text: string) => void): Promise<void> {
    return new Promise((resolve) => {
      const words = fullText.split(' ');
      let index = 0;
      let current = '';

      const interval = setInterval(() => {
        if (index >= words.length) {
          clearInterval(interval);
          resolve();
          return;
        }
        current += (index === 0 ? '' : ' ') + words[index];
        onUpdate(current);
        index++;
      }, 40);
    });
  }
}

import { Component, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../services/chat.service';

@Component({
  selector: 'app-chat-input',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './chat-input.component.html',
  styleUrl: './chat-input.component.scss'
})
export class ChatInputComponent {
  text      = signal('');
  matricule = signal('');
  password  = signal('');

  constructor(public chat: ChatService) {}

  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.send();
    }
  }

  send(): void {
    const value = this.text().trim();
    if (!value || this.chat.isLoading()) return;
    this.chat.sendMessage(value);
    this.text.set('');
  }

  submitAuth(): void {
    const m = this.matricule().trim();
    const p = this.password().trim();
    if (!m || !p || this.chat.isLoading()) return;
    this.chat.sendMessageWithAuth(m, p);
    this.matricule.set('');
    this.password.set('');
  }

  cancelAuth(): void {
    this.matricule.set('');
    this.password.set('');
    this.chat.cancelAuth();
  }
}

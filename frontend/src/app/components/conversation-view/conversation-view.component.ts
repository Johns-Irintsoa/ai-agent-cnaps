import { Component, ElementRef, ViewChild, effect } from '@angular/core';
import { ChatService } from '../../services/chat.service';
import { MessageBubbleComponent } from '../message-bubble/message-bubble.component';
import { TypingIndicatorComponent } from '../typing-indicator/typing-indicator.component';

@Component({
  selector: 'app-conversation-view',
  standalone: true,
  imports: [MessageBubbleComponent, TypingIndicatorComponent],
  templateUrl: './conversation-view.component.html',
  styleUrl: './conversation-view.component.scss'
})
export class ConversationViewComponent {
  @ViewChild('scrollContainer') private container!: ElementRef<HTMLDivElement>;

  constructor(public chat: ChatService) {
    effect(() => {
      this.chat.messages();
      this.chat.isLoading();
      // scroll after DOM update
      setTimeout(() => this.scrollToBottom(), 50);
    });
  }

  private scrollToBottom(): void {
    const el = this.container?.nativeElement;
    if (el) el.scrollTop = el.scrollHeight;
  }

  scrollToMessage(id: string): void {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

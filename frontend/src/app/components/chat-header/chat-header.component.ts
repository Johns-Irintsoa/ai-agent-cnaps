import { Component, EventEmitter, Output } from '@angular/core';
import { ChatService } from '../../services/chat.service';

@Component({
  selector: 'app-chat-header',
  standalone: true,
  templateUrl: './chat-header.component.html',
  styleUrl: './chat-header.component.scss'
})
export class ChatHeaderComponent {
  @Output() toggleSidebar = new EventEmitter<void>();

  constructor(public chat: ChatService) {}
}

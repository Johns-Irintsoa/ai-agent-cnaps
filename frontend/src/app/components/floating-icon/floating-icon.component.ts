import { Component } from '@angular/core';
import { ChatService } from '../../services/chat.service';

@Component({
  selector: 'app-floating-icon',
  standalone: true,
  templateUrl: './floating-icon.component.html',
  styleUrl: './floating-icon.component.scss'
})
export class FloatingIconComponent {
  constructor(public chat: ChatService) {}
}

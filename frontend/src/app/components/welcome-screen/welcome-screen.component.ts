import { Component } from '@angular/core';
import { ChatService } from '../../services/chat.service';

@Component({
  selector: 'app-welcome-screen',
  standalone: true,
  templateUrl: './welcome-screen.component.html',
  styleUrl: './welcome-screen.component.scss'
})
export class WelcomeScreenComponent {
  readonly faqs = [
    'Quels sont les services CNAPS ?',
    'Comment calculer ma retraite ?',
    'Comment s\'inscrire à la CNAPS ?',
    'Quels documents fournir pour une pension ?',
    'Comment contacter la CNAPS ?'
  ];

  constructor(public chat: ChatService) {}

  submit(question: string): void {
    this.chat.sendMessage(question);
  }
}

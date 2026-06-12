import { Component, Input } from '@angular/core';
import { DatePipe } from '@angular/common';
import { Message } from '../../models/message.model';
import { SafeHtmlPipe } from '../../pipes/safe-html.pipe';

@Component({
  selector: 'app-message-bubble',
  standalone: true,
  imports: [DatePipe, SafeHtmlPipe],
  templateUrl: './message-bubble.component.html',
  styleUrl: './message-bubble.component.scss'
})
export class MessageBubbleComponent {
  @Input({ required: true }) message!: Message;
}

import { Injectable, computed } from '@angular/core';
import { ChatService } from './chat.service';
import { SessionEntry } from '../models/message.model';

@Injectable({ providedIn: 'root' })
export class SessionService {
  constructor(private chat: ChatService) {}

  all = computed<SessionEntry[]>(() =>
    this.chat.messages()
      .filter(m => m.role === 'user')
      .map(m => ({ id: m.id, question: m.content, timestamp: m.timestamp }))
  );

  recent = computed<SessionEntry[]>(() => {
    const entries = this.all();
    return entries.slice(-5).reverse();
  });
}

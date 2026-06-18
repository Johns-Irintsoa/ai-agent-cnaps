import { Injectable, signal, computed } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import { Message, RAGResponse, ChatRequest } from '../models/message.model';
import { TypewriterService } from './typewriter.service';

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly apiUrl = '/api/ask';

  messages        = signal<Message[]>([]);
  isOpen          = signal<boolean>(false);
  isLoading       = signal<boolean>(false);
  authRequired    = signal<boolean>(false);
  pendingQuestion = signal<string>('');

  activeView = computed<'welcome' | 'conversation'>(() =>
    this.messages().length === 0 ? 'welcome' : 'conversation'
  );

  constructor(
    private http: HttpClient,
    private typewriter: TypewriterService
  ) {}

  toggleWindow(): void {
    this.isOpen.update(v => !v);
  }

  cancelAuth(): void {
    this.authRequired.set(false);
    this.pendingQuestion.set('');
  }

  async sendMessage(text: string): Promise<void> {
    if (!text.trim() || this.isLoading()) return;
    await this._dispatch({ message: text.trim() }, text.trim());
  }

  async sendMessageWithAuth(matricule: string, password: string): Promise<void> {
    const question = this.pendingQuestion();
    if (!question || this.isLoading()) return;
    this.authRequired.set(false);
    this.pendingQuestion.set('');
    await this._dispatch({ message: question, matricule, password }, question);
  }

  private async _dispatch(payload: ChatRequest, displayText: string): Promise<void> {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: displayText,
      timestamp: new Date(),
    };
    this.messages.update(msgs => [...msgs, userMsg]);
    this.isLoading.set(true);

    try {
      const response = await firstValueFrom(
        this.http.post<RAGResponse>(this.apiUrl, payload)
      );

      if (response?.needs_auth) {
        this.pendingQuestion.set(displayText);
        this.authRequired.set(true);
        this.isLoading.set(false);
        const promptMsg: Message = {
          id: crypto.randomUUID(),
          role: 'bot',
          content: response.answer,
          timestamp: new Date(),
        };
        this.messages.update(msgs => [...msgs, promptMsg]);
        return;
      }

      const answer = response?.answer?.trim();
      if (!answer) throw new Error('empty response');

      const botMsg: Message = {
        id: crypto.randomUUID(),
        role: 'bot',
        content: '',
        timestamp: new Date(),
        metadata: response.metadata,
      };
      this.messages.update(msgs => [...msgs, botMsg]);
      this.isLoading.set(false);

      await this.typewriter.play(answer, (chunk) => {
        this.messages.update(msgs =>
          msgs.map(m => m.id === botMsg.id ? { ...m, content: chunk } : m)
        );
      });

    } catch {
      this.isLoading.set(false);
      this.messages.update(msgs => [...msgs, {
        id: crypto.randomUUID(),
        role: 'bot',
        content: "Je n'ai pas trouvé d'information correspondante. Veuillez reformuler ou contacter directement la CNaPS.",
        timestamp: new Date(),
      }]);
    }
  }

  resetSession(): void {
    this.messages.set([]);
    this.authRequired.set(false);
    this.pendingQuestion.set('');
  }
}

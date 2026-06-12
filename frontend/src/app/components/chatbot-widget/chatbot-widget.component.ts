import { Component, ViewChild, signal } from '@angular/core';
import { ChatService } from '../../services/chat.service';
import { FloatingIconComponent } from '../floating-icon/floating-icon.component';
import { ChatHeaderComponent } from '../chat-header/chat-header.component';
import { WelcomeScreenComponent } from '../welcome-screen/welcome-screen.component';
import { ConversationViewComponent } from '../conversation-view/conversation-view.component';
import { ChatInputComponent } from '../chat-input/chat-input.component';
import { SidebarComponent } from '../sidebar/sidebar.component';

@Component({
  selector: 'app-chatbot-widget',
  standalone: true,
  imports: [
    FloatingIconComponent,
    ChatHeaderComponent,
    WelcomeScreenComponent,
    ConversationViewComponent,
    ChatInputComponent,
    SidebarComponent
  ],
  templateUrl: './chatbot-widget.component.html',
  styleUrl: './chatbot-widget.component.scss'
})
export class ChatbotWidgetComponent {
  @ViewChild(ConversationViewComponent) private conversationView?: ConversationViewComponent;

  sidebarOpen = signal(false);

  constructor(public chat: ChatService) {}

  toggleSidebar(): void {
    this.sidebarOpen.update(v => !v);
  }

  resetAndClose(): void {
    this.chat.resetSession();
  }

  handleNavigateTo(id: string): void {
    this.sidebarOpen.set(false);
    setTimeout(() => this.conversationView?.scrollToMessage(id), 50);
  }
}

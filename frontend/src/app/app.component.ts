import { Component } from '@angular/core';
import { ChatbotWidgetComponent } from './components/chatbot-widget/chatbot-widget.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [ChatbotWidgetComponent],
  template: `<app-chatbot-widget />`
})
export class AppComponent {}

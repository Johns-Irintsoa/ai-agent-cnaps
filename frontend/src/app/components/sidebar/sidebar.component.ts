import { Component, EventEmitter, Output, signal } from '@angular/core';
import { SessionService } from '../../services/session.service';

@Component({
  selector: 'app-sidebar',
  standalone: true,
  templateUrl: './sidebar.component.html',
  styleUrl: './sidebar.component.scss'
})
export class SidebarComponent {
  @Output() newChat = new EventEmitter<void>();
  @Output() navigateTo = new EventEmitter<string>();

  activeTab = signal<'recent' | 'all'>('recent');

  constructor(public session: SessionService) {}

  scrollTo(id: string): void {
    this.navigateTo.emit(id);
  }
}

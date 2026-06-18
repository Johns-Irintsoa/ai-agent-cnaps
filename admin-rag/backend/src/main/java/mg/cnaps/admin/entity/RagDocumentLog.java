package mg.cnaps.admin.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity
@Table(name = "RAG_DOCUMENT_LOG")
@Getter
@Setter
@NoArgsConstructor
public class RagDocumentLog {

    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "log_seq")
    @SequenceGenerator(name = "log_seq", sequenceName = "RAG_DOCUMENT_LOG_SEQ", allocationSize = 1)
    @Column(name = "ID")
    private Long id;

    @Column(name = "DOCUMENT_ID", length = 36)
    private String documentId;

    @Column(name = "STEP", nullable = false, length = 50)
    private String step;

    @Column(name = "MESSAGE", length = 1000)
    private String message;

    @Column(name = "CREATED_AT")
    private LocalDateTime createdAt;

    @PrePersist
    void onCreate() {
        createdAt = LocalDateTime.now();
    }

    public RagDocumentLog(String documentId, String step, String message) {
        this.documentId = documentId;
        this.step = step;
        this.message = message;
    }
}

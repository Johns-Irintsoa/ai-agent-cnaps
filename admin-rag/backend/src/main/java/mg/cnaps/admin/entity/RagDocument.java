package mg.cnaps.admin.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity
@Table(name = "RAG_DOCUMENT", indexes = {
    @Index(name = "idx_rag_doc_status", columnList = "STATUS"),
    @Index(name = "idx_rag_doc_type",   columnList = "FILE_TYPE"),
    @Index(name = "idx_rag_doc_user",   columnList = "UPLOADED_BY")
})
@Getter
@Setter
@NoArgsConstructor
public class RagDocument {

    @Id
    @Column(name = "ID", length = 36)
    private String id;

    @Column(name = "FILENAME", nullable = false, length = 255)
    private String filename;

    @Enumerated(EnumType.STRING)
    @Column(name = "FILE_TYPE", nullable = false, length = 10)
    private FileType fileType;

    @Column(name = "FILE_SIZE", nullable = false)
    private long fileSize;

    @Column(name = "FILE_PATH", length = 500)
    private String filePath;

    @Column(name = "SOURCE_URL", length = 1000)
    private String sourceUrl;

    @Enumerated(EnumType.STRING)
    @Column(name = "STATUS", nullable = false, length = 20)
    private DocumentStatus status;

    @Column(name = "ERROR_MESSAGE", length = 2000)
    private String errorMessage;

    @Column(name = "CHUNK_COUNT")
    private int chunkCount = 0;

    @Column(name = "UPLOADED_BY", nullable = false, length = 100)
    private String uploadedBy;

    @Column(name = "IMPORTED_AT")
    private LocalDateTime importedAt;

    @Column(name = "PROCESSED_AT")
    private LocalDateTime processedAt;

    @Column(name = "UPDATED_AT")
    private LocalDateTime updatedAt;

    @PrePersist
    void onCreate() {
        importedAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
}

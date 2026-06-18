package mg.cnaps.admin.repository;

import mg.cnaps.admin.entity.RagDocumentLog;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface DocumentLogRepository extends JpaRepository<RagDocumentLog, Long> {

    List<RagDocumentLog> findByDocumentIdOrderByCreatedAtAsc(String documentId);
}

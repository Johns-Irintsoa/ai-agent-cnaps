package mg.cnaps.admin.repository;

import mg.cnaps.admin.entity.DocumentStatus;
import mg.cnaps.admin.entity.RagDocument;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import java.util.List;

public interface DocumentRepository extends JpaRepository<RagDocument, String> {

    long countByStatus(DocumentStatus status);

    // ─── Pagination native Oracle 11g (ROWNUM) ────────────────────────────────
    // Oracle 11g ne supporte pas FETCH FIRST n ROWS ONLY (12c+).
    // Pattern : double sous-requête ROWNUM pour simuler OFFSET + LIMIT.

    @Query(value = """
            SELECT ID,CHUNK_COUNT,ERROR_MESSAGE,FILE_PATH,FILE_SIZE,FILE_TYPE,
                   FILENAME,IMPORTED_AT,PROCESSED_AT,SOURCE_URL,STATUS,UPDATED_AT,UPLOADED_BY
            FROM (
              SELECT t.*,ROWNUM rn FROM (
                SELECT * FROM RAG_DOCUMENT ORDER BY IMPORTED_AT DESC
              ) t WHERE ROWNUM <= :endRow
            ) WHERE rn > :startRow
            """, nativeQuery = true)
    List<RagDocument> findAllNative(@Param("startRow") int startRow,
                                    @Param("endRow")   int endRow);

    @Query(value = """
            SELECT ID,CHUNK_COUNT,ERROR_MESSAGE,FILE_PATH,FILE_SIZE,FILE_TYPE,
                   FILENAME,IMPORTED_AT,PROCESSED_AT,SOURCE_URL,STATUS,UPDATED_AT,UPLOADED_BY
            FROM (
              SELECT t.*,ROWNUM rn FROM (
                SELECT * FROM RAG_DOCUMENT WHERE STATUS = :status ORDER BY IMPORTED_AT DESC
              ) t WHERE ROWNUM <= :endRow
            ) WHERE rn > :startRow
            """, nativeQuery = true)
    List<RagDocument> findByStatusNative(@Param("status")   String status,
                                          @Param("startRow") int    startRow,
                                          @Param("endRow")   int    endRow);

    @Query(value = "SELECT COUNT(*) FROM RAG_DOCUMENT WHERE STATUS = :status",
           nativeQuery = true)
    long countByStatusNative(@Param("status") String status);

    @Query(value = """
            SELECT ID,CHUNK_COUNT,ERROR_MESSAGE,FILE_PATH,FILE_SIZE,FILE_TYPE,
                   FILENAME,IMPORTED_AT,PROCESSED_AT,SOURCE_URL,STATUS,UPDATED_AT,UPLOADED_BY
            FROM (
              SELECT t.*,ROWNUM rn FROM (
                SELECT * FROM RAG_DOCUMENT
                WHERE UPPER(FILENAME) LIKE UPPER('%'||:filename||'%')
                ORDER BY IMPORTED_AT DESC
              ) t WHERE ROWNUM <= :endRow
            ) WHERE rn > :startRow
            """, nativeQuery = true)
    List<RagDocument> findByFilenameNative(@Param("filename") String filename,
                                            @Param("startRow") int    startRow,
                                            @Param("endRow")   int    endRow);

    @Query(value = "SELECT COUNT(*) FROM RAG_DOCUMENT WHERE UPPER(FILENAME) LIKE UPPER('%'||:filename||'%')",
           nativeQuery = true)
    long countByFilenameNative(@Param("filename") String filename);

    @Query(value = """
            SELECT ID,CHUNK_COUNT,ERROR_MESSAGE,FILE_PATH,FILE_SIZE,FILE_TYPE,
                   FILENAME,IMPORTED_AT,PROCESSED_AT,SOURCE_URL,STATUS,UPDATED_AT,UPLOADED_BY
            FROM RAG_DOCUMENT
            WHERE ROWNUM = 1
            ORDER BY UPDATED_AT DESC
            """, nativeQuery = true)
    List<RagDocument> findTopDocument();
}

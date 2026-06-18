package mg.cnaps.admin.dto;

public record DashboardStats(
        long totalIndexed,
        long totalSuccess,
        long totalError,
        long totalProcessing,
        long totalPending,
        String lastUpdatedAt,
        String lastUpdatedFilename
) {}

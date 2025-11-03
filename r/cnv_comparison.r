# Packages ---------------------------------------------------------------
pkgs <- c("readxl","janitor","dplyr","tidyr","stringr","purrr","readr",
          "GenomicRanges","IRanges","GenomeInfoDb","tibble","ggplot2")
to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, repos="https://cloud.r-project.org")
invisible(lapply(pkgs, library, character.only=TRUE))

# Paths & thresholds -----------------------------------------------------
sample_sheet_path <- "./Sample_Sheet.xlsx"
igv_dir <- "./EPIC/bin/"   # change if your .igv files are elsewhere
cnr_dir <- "./cnr/"        # change if your .cnr files are elsewhere

build <- "hg19"            # "hg19" or "hg38"
keep_sex <- FALSE          # set TRUE to include X/Y arms

# Distinct thresholds per method for class calls based on *mean arm log2*
cnr_gain_thr <- 0.50; cnr_loss_thr <- 0.50
igv_gain_thr <- 0.50; igv_loss_thr <- 0.50

# Centromeres ------------------------------------------------------------
centromeres_hg19 <- tribble(
  ~chrom, ~cent_start, ~cent_end,
  "chr1",121535434,124535434,"chr2",92326171,95326171,
  "chr3",90504854,93504854,"chr4",49660117,52660117,
  "chr5",46405641,49405641,"chr6",58830166,61830166,
  "chr7",58054331,61054331,"chr8",43838887,46838887,
  "chr9",47367679,50367679,"chr10",39254935,42254935,
  "chr11",51644205,54644205,"chr12",34856694,37856694,
  "chr13",16000000,19000000,"chr14",16000000,19000000,
  "chr15",17000000,20000000,"chr16",35335801,38335801,
  "chr17",22263006,25263006,"chr18",15460898,18460898,
  "chr19",24681782,27681782,"chr20",26369569,29369569,
  "chr21",11288129,14288129,"chr22",13000000,16000000,
  "chrX",58632012,61632012,"chrY",10316945,12316945
) |> set_names(c("chrom","cent_start","cent_end"))
centromeres_hg38 <- tribble(
  ~chrom, ~cent_start, ~cent_end,
  "chr1",122026459,125184587,"chr2",92188145,94090557,
  "chr3",90772458,93655574,"chr4",49712061,51743951,
  "chr5",46485900,50059807,"chr6",58553888,59829934,
  "chr7",58169618,61055216,"chr8",44033744,45877265,
  "chr9",43243682,45518558,"chr10",39686683,41593521,
  "chr11",51078349,54425074,"chr12",34769408,37185252,
  "chr13",16000000,18000000,"chr14",16000000,18000000,
  "chr15",17000000,20000000,"chr16",36311159,38280682,
  "chr17",22700000,27400000,"chr18",15400000,18300000,
  "chr19",24400000,28100000,"chr20",26400000,30000000,
  "chr21",10900000,14300000,"chr22",13700000,19000000,
  "chrX",58632012,62412542,"chrY",10316945,10544039
) |> set_names(c("chrom","cent_start","cent_end"))
centromeres <- if (build=="hg19") centromeres_hg19 else centromeres_hg38

# Sample sheet -----------------------------------------------------------
ss <- readxl::read_excel(sample_sheet_path) |> janitor::clean_names()
stopifnot(all(c("sentrix_id","barcode") %in% names(ss)))
if (!"sample" %in% names(ss)) ss$sample <- NA_character_
ss <- ss |>
  mutate(sample_id = dplyr::coalesce(as.character(.data$sample), paste0("S_", row_number())))

# Results containers -----------------------------------------------------
summ_rows <- list()
discord_rows <- list()

# Iterate samples --------------------------------------------------------
for (i in seq_len(nrow(ss))) {
  sentrix_id <- ss$sentrix_id[i]
  barcode    <- ss$barcode[i]
  sample_id  <- ss$sample_id[i]
  
  # Find files
  igv_path <- list.files(igv_dir, pattern=paste0(".*", stringr::fixed(sentrix_id), ".*\\.igv$"),
                         full.names=TRUE, recursive=TRUE, ignore.case=TRUE)
  cnr_path <- list.files(cnr_dir, pattern=paste0(".*", stringr::fixed(barcode), ".*\\.cnr$"),
                         full.names=TRUE, recursive=TRUE, ignore.case=TRUE)
  if (length(igv_path)==0 || length(cnr_path)==0) {
    summ_rows[[length(summ_rows)+1]] <- tibble(
      sample_id, sentrix_id, barcode,
      n_arms=NA_integer_, n_mismatched=NA_integer_, prop_match=NA_real_,
      r_meanlog2_per_arm=NA_real_, kappa=NA_real_,
      igv_path=ifelse(length(igv_path)==0, NA, igv_path[1]),
      cnr_path=ifelse(length(cnr_path)==0, NA, cnr_path[1]),
      status="missing_file"
    )
    next
  }
  igv_path <- igv_path[1]; cnr_path <- cnr_path[1]
  
  # Read IGV (choose the Sentrix column if present; else last numeric column)
  igv_raw <- suppressMessages(readr::read_tsv(igv_path, show_col_types = FALSE))
  stopifnot(all(c("Chromosome","Start","End") %in% names(igv_raw)))
  igv_num_cols <- names(igv_raw)[sapply(igv_raw, is.numeric)]
  igv_num_cols <- setdiff(igv_num_cols, c("Start","End"))
  igv_col <- if (sentrix_id %in% names(igv_raw)) sentrix_id else if (length(igv_num_cols)) igv_num_cols[length(igv_num_cols)] else NA_character_
  if (is.na(igv_col)) {
    summ_rows[[length(summ_rows)+1]] <- tibble(
      sample_id, sentrix_id, barcode,
      n_arms=NA_integer_, n_mismatched=NA_integer_, prop_match=NA_real_,
      r_meanlog2_per_arm=NA_real_, kappa=NA_real_,
      igv_path, cnr_path, status="no_log2_col_in_igv"
    )
    next
  }
  igv_df <- igv_raw |>
    transmute(
      chrom = as.character(Chromosome),
      chrom = ifelse(startsWith(chrom, "chr"), chrom, paste0("chr", chrom)),
      start = as.integer(Start),
      end   = as.integer(End),
      log2  = as.numeric(.data[[igv_col]])
    )
  
  # Read CNR
  cnr_raw <- suppressMessages(readr::read_tsv(cnr_path, show_col_types = FALSE))
  stopifnot(all(c("chromosome","start","end","log2") %in% names(cnr_raw)))
  cnr_df <- cnr_raw |>
    transmute(
      chrom = as.character(chromosome),
      chrom = ifelse(startsWith(chrom, "chr"), chrom, paste0("chr", chrom)),
      start = as.integer(start),
      end   = as.integer(end),
      log2  = as.numeric(log2)
    )
  
  # Build arm ranges covered by both methods
  chroms <- intersect(unique(cnr_df$chrom), unique(igv_df$chrom))
  if (!keep_sex) chroms <- setdiff(chroms, c("chrX","chrY","X","Y"))
  if (length(chroms)==0) {
    summ_rows[[length(summ_rows)+1]] <- tibble(
      sample_id, sentrix_id, barcode,
      n_arms=0L, n_mismatched=NA_integer_, prop_match=NA_real_,
      r_meanlog2_per_arm=NA_real_, kappa=NA_real_,
      igv_path, cnr_path, status="no_common_chroms"
    )
    next
  }
  
  arms_tbl <- tibble()
  for (ch in chroms) {
    cmin <- min(c(cnr_df$start[cnr_df$chrom==ch], igv_df$start[igv_df$chrom==ch]), na.rm=TRUE)
    cmax <- max(c(cnr_df$end  [cnr_df$chrom==ch], igv_df$end  [igv_df$chrom==ch]), na.rm=TRUE)
    ce <- centromeres |> filter(.data$chrom==ch)
    if (nrow(ce)==0) next
    arms_tbl <- bind_rows(
      arms_tbl,
      tibble(chrom=ch, arm="p", start=cmin, end=ce$cent_start - 1L),
      tibble(chrom=ch, arm="q", start=ce$cent_end + 1L, end=cmax)
    )
  }
  arms_tbl <- arms_tbl |> filter(.data$start < .data$end)
  if (nrow(arms_tbl)==0) {
    summ_rows[[length(summ_rows)+1]] <- tibble(
      sample_id, sentrix_id, barcode,
      n_arms=0L, n_mismatched=NA_integer_, prop_match=NA_real_,
      r_meanlog2_per_arm=NA_real_, kappa=NA_real_,
      igv_path, cnr_path, status="no_arms"
    )
    next
  }
  arms_gr <- GRanges(seqnames=arms_tbl$chrom, ranges=IRanges(start=arms_tbl$start, end=arms_tbl$end), arm=arms_tbl$arm)
  
  # Length-weighted mean log2 per arm (CNVkit)
  cnr_gr <- GRanges(seqnames=cnr_df$chrom, ranges=IRanges(cnr_df$start, cnr_df$end))
  mcols(cnr_gr)$log2 <- cnr_df$log2
  cnr_mean <- tibble()
  for (j in seq_along(arms_gr)) {
    a <- arms_gr[j]
    ov <- findOverlaps(cnr_gr, a)
    if (!length(ov)) {
      cnr_mean <- bind_rows(cnr_mean, tibble(chrom=as.character(seqnames(a)), arm=as.character(mcols(a)$arm), mean_log2_cnr=NA_real_))
      next
    }
    sub <- cnr_gr[queryHits(ov)]
    ov_int <- pintersect(sub, rep(a, length(sub)))
    w <- width(ov_int); w[is.na(w)] <- 0L
    mn <- if (sum(w)>0) stats::weighted.mean(mcols(sub)$log2, w=w, na.rm=TRUE) else NA_real_
    cnr_mean <- bind_rows(cnr_mean, tibble(chrom=as.character(seqnames(a)), arm=as.character(mcols(a)$arm), mean_log2_cnr=mn))
  }
  
  # Length-weighted mean log2 per arm (IGV)
  igv_gr <- GRanges(seqnames=igv_df$chrom, ranges=IRanges(igv_df$start, igv_df$end))
  mcols(igv_gr)$log2 <- igv_df$log2
  igv_mean <- tibble()
  for (j in seq_along(arms_gr)) {
    a <- arms_gr[j]
    ov <- findOverlaps(igv_gr, a)
    if (!length(ov)) {
      igv_mean <- bind_rows(igv_mean, tibble(chrom=as.character(seqnames(a)), arm=as.character(mcols(a)$arm), mean_log2_igv=NA_real_))
      next
    }
    sub <- igv_gr[queryHits(ov)]
    ov_int <- pintersect(sub, rep(a, length(sub)))
    w <- width(ov_int); w[is.na(w)] <- 0L
    mn <- if (sum(w)>0) stats::weighted.mean(mcols(sub)$log2, w=w, na.rm=TRUE) else NA_real_
    igv_mean <- bind_rows(igv_mean, tibble(chrom=as.character(seqnames(a)), arm=as.character(mcols(a)$arm), mean_log2_igv=mn))
  }
  
  # Merge and call per arm
  arm_means <- inner_join(cnr_mean, igv_mean, by=c("chrom","arm")) |> filter(is.finite(mean_log2_cnr) & is.finite(mean_log2_igv))
  if (nrow(arm_means)==0) {
    summ_rows[[length(summ_rows)+1]] <- tibble(
      sample_id, sentrix_id, barcode,
      n_arms=0L, n_mismatched=NA_integer_, prop_match=NA_real_,
      r_meanlog2_per_arm=NA_real_, kappa=NA_real_,
      igv_path, cnr_path, status="no_overlap_bins"
    )
    next
  }
  
  arm_means <- arm_means |>
    mutate(
      cnr_call = ifelse(mean_log2_cnr >= cnr_gain_thr, "gain",
                        ifelse(mean_log2_cnr <= -abs(cnr_loss_thr), "loss", "neutral")),
      igv_call = ifelse(mean_log2_igv >= igv_gain_thr, "gain",
                        ifelse(mean_log2_igv <= -abs(igv_loss_thr), "loss", "neutral")),
      region_id = paste0(chrom, arm)
    )
  
  # Confusion, correlation, kappa (simple)
  lev <- c("gain","loss","neutral")
  tab <- table(factor(arm_means$cnr_call, levels=lev), factor(arm_means$igv_call, levels=lev))
  n <- sum(tab); p0 <- sum(diag(tab))/n
  pr <- rowSums(tab)/n; pc <- colSums(tab)/n
  pe <- sum(pr*pc); kappa_val <- if (1-pe==0) NA_real_ else (p0 - pe)/(1 - pe)
  
  r_arm <- suppressWarnings(stats::cor(arm_means$mean_log2_cnr, arm_means$mean_log2_igv, use="pairwise"))
  mismatched <- arm_means$region_id[arm_means$cnr_call != arm_means$igv_call]
  
  summ_rows[[length(summ_rows)+1]] <- tibble(
    sample_id, sentrix_id, barcode,
    n_arms = nrow(arm_means),
    n_mismatched = length(mismatched),
    prop_match = ifelse(nrow(arm_means)>0, mean(arm_means$cnr_call == arm_means$igv_call), NA_real_),
    r_meanlog2_per_arm = r_arm,
    kappa = kappa_val,
    igv_path, cnr_path,
    status = "ok"
  )
  
  if (length(mismatched)) {
    discord_rows[[length(discord_rows)+1]] <- tibble(sample_id, region_id = mismatched)
  }
}

# Summaries --------------------------------------------------------------
summ_tbl <- bind_rows(summ_rows) |>
  mutate(status = ifelse(is.na(status), "ok", status)) |>
  arrange(desc(status=="ok"), desc(prop_match), n_mismatched)

mean(na.omit(summ_tbl$prop_match))

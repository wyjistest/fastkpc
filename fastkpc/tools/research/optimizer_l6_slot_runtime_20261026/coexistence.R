# Reuse frozen canonical gates; label all elapsed values as diagnostic only.
path <- "fastkpc/tools/research/optimizer_l4_slot_runtime_20261024/run.R"
lines <- readLines(path,warn=FALSE)
old <- "GPU1 isolated incremental source-closure development; not full rebuild, clean producer or promotion"
stopifnot(sum(grepl(old,lines,fixed=TRUE))==1L)
lines <- sub(old,"GPU1 isolated L6 replacement coexistence diagnostic; not performance evidence",lines,fixed=TRUE)
eval(parse(text=lines),envir=.GlobalEnv)

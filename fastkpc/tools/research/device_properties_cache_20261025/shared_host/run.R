# Preserve the frozen numerical runner; change only its evidence-class label.
# The enclosing protocol binds this wrapper and the original runner separately.
original <- "fastkpc/tools/research/device_properties_cache_20261025/run.R"
script_lines <- readLines(original, warn=FALSE)
isolated_label <- "GPU1 isolated incremental source-closure development; not full rebuild, clean producer or promotion"
shared_label <- "GPU1 exclusive, shared-host supplemental development; not isolated performance qualification, clean producer or promotion"
stopifnot(sum(grepl(isolated_label, script_lines, fixed=TRUE))==1L)
script_lines <- sub(isolated_label, shared_label, script_lines, fixed=TRUE)
eval(parse(text=script_lines), envir=.GlobalEnv)

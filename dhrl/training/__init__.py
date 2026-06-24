"""Training — the two-stage curriculum (paper §5.2): stage 1 pre-trains the lower
operator single-agent then freezes it; stage 2 trains UL+ML with IPPO over the frozen LL."""

# Trasform the temporal [(h*kt) x 1] vector into a [h x kt] matrix
# See also: hmat2vec()
vec2hmat <- function(vec, h, kset) {
  m <- max(kset)
  i <- rep(rep(1:h, length(kset)), rep(m / kset, each = h))
  matrix(vec[order(i)], nrow = h, byrow = T)
}

# Trasform the [h x kt] matrix into a temporal [(h*kt) x 1] vector
# See also: mat2hmat()
hmat2vec <- function(hmat, h, kset) {
  m <- max(kset)
  i <- rep(1:sum(m / kset), h)
  it <- rep(rep(m / kset, m / kset), h)
  ih <- rep(1:h, each = sum(m / kset))
  out <- as.vector(t(hmat))[order(it, ih, i)]
  names_vec <- namesTE(kset = kset, h = h)
  setNames(out, names_vec)
}

# Build a named vector to specify k and h position
namesTE <- function(kset, h) {
  m <- max(kset)
  seqk <- h * (m / kset)
  paste0("k-", rep(kset, seqk), " h-", Reduce("c", sapply(seqk, seq_len)))
}

# Trasform the cross-temporal [n x (h*kt)] matrix into a [h x (n*kt)] matrix
# See also: hmat2mat()
mat2hmat <- function(mat, h, kset, n) {
  m <- max(kset)
  i <- rep(rep(rep(1:h, length(kset)), rep(m / kset, each = h)), n)
  vec <- as.vector(t(mat))
  matrix(vec[order(i)], nrow = h, byrow = T)
}

# Trasform the [h x (n*kt)] matrix into a cross-temporal [n x (h*kt)] matrix
# See also: mat2hmat()
hmat2mat <- function(hmat, h, kset, n) {
  m <- max(kset)
  i <- rep(1:sum(m / kset), h * n)
  it <- rep(rep(m / kset, m / kset), h * n)
  ih <- rep(1:h, each = n * sum(m / kset))
  out <- matrix(as.vector(t(hmat))[order(it, ih, i)], nrow = n)
  colnames(out) <- namesTE(kset = kset, h = h)
  out
}

# Split cross-temporal matrix in a temporal list
mat2list <- function(mat, kset) {
  m <- max(kset)
  h <- NCOL(mat) / sum(kset)
  kid <- rep(kset, h * m / kset)
  split.data.frame(t(mat), kid)[as.character(kset)]
}

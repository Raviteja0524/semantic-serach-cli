import click
from pathlib import Path
from src.embedder import Embedder, EmbedderConfig
from src.vector_store import VectorStore

DEFAULT_STORE = "data/store.npz"


@click.group()
def cli():
    """Semantic Search CLI — search documents by meaning, not keywords."""
    pass


@cli.command()
@click.option("--file", "filepath", required=True, type=click.Path(exists=True), help="Path to the text file to index.")
@click.option("--store", default=DEFAULT_STORE, show_default=True, help="Where to save the vector store.")
def index(filepath, store):
    """Read a file and index each line as a searchable document."""
    lines = [line.strip() for line in Path(filepath).read_text().splitlines() if line.strip()]

    if not lines:
        click.echo("No content found in file.")
        return

    click.echo(f"Indexing {len(lines)} documents from '{filepath}'...")

    embedder = Embedder(EmbedderConfig())
    vector_store = VectorStore()

    with click.progressbar(lines, label="Embedding") as bar:
        for i, line in enumerate(bar):
            embedding = embedder.embed(line)
            vector_store.add(id=f"doc_{i}", text=line, embedding=embedding)

    Path(store).parent.mkdir(parents=True, exist_ok=True)
    vector_store.save(store)
    click.echo(f"\nSaved {len(lines)} documents to '{store}'.")


@cli.command()
@click.argument("query")
@click.option("--store", default=DEFAULT_STORE, show_default=True, help="Path to the vector store to search.")
@click.option("--top-k", default=3, show_default=True, help="Number of results to return.")
def search(query, store, top_k):
    """Search the indexed documents using a natural language query."""
    store_path = Path(store)
    if not store_path.exists():
        click.echo(f"No vector store found at '{store}'. Run `index` first.")
        return

    embedder = Embedder(EmbedderConfig())
    vector_store = VectorStore()
    vector_store.load(store)

    query_embedding = embedder.embed(query)
    results = vector_store.search(query_embedding, top_k=top_k)

    click.echo(f"\nTop {top_k} results for: \"{query}\"\n")
    click.echo("-" * 60)
    for rank, result in enumerate(results, start=1):
        click.echo(f"#{rank}  Score: {result.score:.4f}")
        click.echo(f"    {result.text}")
        click.echo()


if __name__ == "__main__":
    cli()

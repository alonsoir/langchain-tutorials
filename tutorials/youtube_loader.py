from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader, TextLoader
from langchain_openai import OpenAI


def main():
    print("main")
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="stuff", verbose=True)
    url = "https://www.youtube.com/watch?v=mpgr0cdW3VE"
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    # Descarga los subtítulos en español
    subtitles = loader.translation
    if subtitles is not None:
        # Carga los subtítulos como un documento de texto
        text_loader = TextLoader(file_path=subtitles, autodetect_encoding=True)
        if text_loader is not None:
            # Carga la cadena de resumen
            summarize_chain = load_summarize_chain(
                llm=llm, chain_type="stuff", verbose=True
            )
            # Resume el video usando los subtítulos
            summary = summarize_chain(text_loader)
            print(summary)
    else:
        print(f"El video {url} no tiene subtitulos que pueda cargar...")

    result = loader.load()

    print(
        f"There is a video in url {url} from the author {result[0].metadata['author']} that is {result[0].metadata['length']} seconds long"
    )
    print("")
    if result is not None:
        print(f"content is: {result[0].page_content}")
        chain_result = chain.run(result)
        print(f"chain_result is {chain_result}")
    another_url = "https://www.youtube.com/shorts/Y6S9csg1-Wo"
    loader = YoutubeLoader.from_youtube_url(another_url, add_video_info=True)
    result = loader.load()
    print(
        f"There is a video in url {another_url} from the author {result[0].metadata['author']} that is {result[0].metadata['length']} seconds long"
    )
    chain = load_summarize_chain(llm, chain_type="stuff", verbose=False)
    another_chain_result = chain.run(result)
    print(f"another_chain_result is: {another_chain_result}")


if __name__ == "__main__":
    load_dotenv()
    main()

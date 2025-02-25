from dotenv import load_dotenv
from fledge.connect import connnect_fedn_client, parse_args


load_dotenv()


def main():
    print("[*] Connecting Edge Client:")
    fedn_args = parse_args()
    connnect_fedn_client(
        "test_client_0",
        api_url=fedn_args.api_url,
        token=fedn_args.token,
        api_port=fedn_args.api_port,
    )


if __name__ == "__main__":
    main()

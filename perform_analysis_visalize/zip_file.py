import os
import sys
import zipfile

def zip_folder(folder_path, zip_path=None):
    """
    压缩整个文件夹为 zip 文件。

    :param folder_path: 要压缩的文件夹路径
    :param zip_path: 生成的 zip 文件路径（可选，不传则用 folder_path 同名）
    """
    folder_path = os.path.abspath(folder_path)

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"{folder_path} 不是一个有效的文件夹")

    # 如果没有指定 zip 文件名，就用文件夹名 + .zip
    if zip_path is None:
        parent_dir = os.path.dirname(folder_path)
        folder_name = os.path.basename(folder_path.rstrip(os.sep))
        zip_path = os.path.join(parent_dir, f"{folder_name}.zip")

    # 创建 zip 文件
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # arcname 是 zip 里面保存的相对路径
                arcname = os.path.relpath(file_path, start=folder_path)
                zf.write(file_path, arcname)

    return zip_path


def main():
    if len(sys.argv) < 2:
        print("用法: python zip_folder.py <folder_path> [zip_path]")
        sys.exit(1)

    folder_path = sys.argv[1]
    zip_path = sys.argv[2] if len(sys.argv) >= 3 else None

    try:
        result = zip_folder(folder_path, zip_path)
        print(f"压缩完成: {result}")
    except Exception as e:
        print(f"压缩失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#/bin/bash
cd mstream/
make clean
make
cd ../

if [ $1 == "CentralParkNYC-2021-01-27-2021-02-06" ]; then
  echo "CentralParkNYC"
  echo "Vanilla MSTREAM"
  mstream/mstream -t 'data/CentralParkNYCtime.txt' -n 'data/CentralParkNYCnumeric.txt' -c 'data/CentralParkNYCcateg.txt' -o 'data/CentralParkNYC_score.txt' -a 0.8
  python3 results.py --label 'data/CentralParkNYC_label.txt' --scores 'data/CentralParkNYC_score.txt'
  : '
  echo "MSTREAM-PCA"
  python3 pca.py --input 'data/CentralParkNYCnumeric.txt'
  mstream/mstream -t 'data/CentralParkNYCtime.txt' -n 'pca.txt' -c 'data/CentralParkNYCcateg.txt' -o 'pcascore.txt' -a 0.8
  python3 results.py --label 'data/CentralParkNYC_label.txt' --scores 'pcascore.txt'

  echo "MSTREAM-IB"
  python3 ib.py --input 'data/CentralParkNYCnumeric.txt' --inputdim 34 --label 'data/CentralParkNYC_label.txt' --lr 0.01 --numEpochs 100
  echo "MSTREAM-IB-2"
  mstream/mstream -t 'data/CentralParkNYCtime.txt' -n 'ib.txt' -c 'data/CentralParkNYCcateg.txt' -o 'ibscore.txt' -a 0.8
  python3 results.py --label 'data/CentralParkNYC_label.txt' --scores 'ibscore.txt'

  echo "MSTREAM-AE"
  python3 ae.py --input 'data/CentralParkNYCnumeric.txt' --inputdim 34 --lr 0.01 --numEpochs 100
  mstream/mstream -t 'data/CentralParkNYCtime.txt' -n 'ae.txt' -c 'data/CentralParkNYCcateg.txt' -o 'aescore.txt' -a 0.8
  python3 results.py --label 'data/CentralParkNYC_label.txt' --scores 'aescore.txt'
  '
fi

if [ $1 == "CentralParkNYC" ]; then
  echo "CentralParkNYC"
  echo "Vanilla MSTREAM"
  mstream/mstream -t 'data/CentralParkNYC_time.txt' -n 'data/CentralParkNYC_numeric.txt' -c 'data/CentralParkNYC_categ.txt' -o 'data/CentralParkNYC_score.txt' -a 0.8
  python3 results.py --label 'data/CentralParkNYC_label.txt' --scores 'data/CentralParkNYC_score.txt'
fi

if [ $1 == "KDD" ]; then
  echo "KDD"
  echo "Vanilla MSTREAM"
  mstream/mstream -t 'data/kddtime.txt' -n 'data/kddnumeric.txt' -c 'data/kddcateg.txt' -o 'score.txt' -a 0.8
  python3 results.py --label 'data/kdd_label.txt' --scores 'score.txt'

  echo "MSTREAM-PCA"
  python3 pca.py --input 'data/kddnumeric.txt'
  mstream/mstream -t 'data/kddtime.txt' -n 'pca.txt' -c 'data/kddcateg.txt' -o 'pcascore.txt' -a 0.8
  python3 results.py --label 'data/kdd_label.txt' --scores 'pcascore.txt'

  echo "MSTREAM-IB"
  python3 ib.py --input 'data/kddnumeric.txt' --inputdim 34 --label 'data/kdd_label.txt' --lr 0.01 --numEpochs 100
  echo "MSTREAM-IB-2"
  mstream/mstream -t 'data/kddtime.txt' -n 'ib.txt' -c 'data/kddcateg.txt' -o 'ibscore.txt' -a 0.8
  python3 results.py --label 'data/kdd_label.txt' --scores 'ibscore.txt'

  echo "MSTREAM-AE"
  python3 ae.py --input 'data/kddnumeric.txt' --inputdim 34 --lr 0.01 --numEpochs 100
  mstream/mstream -t 'data/kddtime.txt' -n 'ae.txt' -c 'data/kddcateg.txt' -o 'aescore.txt' -a 0.8
  python3 results.py --label 'data/kdd_label.txt' --scores 'aescore.txt'
fi

if [ $1 == "UNSW" ]; then
  echo "UNSW"
  echo "Vanilla MSTREAM"
  mstream/mstream -t 'data/unswtime.txt' -n 'data/unswnumeric.txt' -c 'data/unswcateg.txt' -o 'score.txt' -a 0.4
  python3 results.py --label 'data/unsw_label.txt' --scores 'score.txt'

  echo "MSTREAM-PCA"
  python3 pca.py --input 'data/unswnumeric.txt'
  mstream/mstream -t 'data/unswtime.txt' -n 'pca.txt' -c 'data/unswcateg.txt' -o 'pcascore.txt' -a 0.4
  python3 results.py --label 'data/unsw_label.txt' --scores 'pcascore.txt'

  echo "MSTREAM-IB"
  python3 ib.py --input 'data/unswnumeric.txt' --inputdim 39 --label 'data/unsw_label.txt' --lr 0.01 --numEpochs 100
  mstream/mstream -t 'data/unswtime.txt' -n 'ib.txt' -c 'data/unswcateg.txt' -o 'ibscore.txt' -a 0.4
  python3 results.py --label 'data/unsw_label.txt' --scores 'ibscore.txt'

  echo "MSTREAM-AE"
  python3 ae.py --input 'data/unswnumeric.txt' --inputdim 39 --lr 0.01 --numEpochs 100
  mstream/mstream -t 'data/unswtime.txt' -n 'ae.txt' -c 'data/unswcateg.txt' -o 'aescore.txt' -a 0.4
  python3 results.py --label 'data/unsw_label.txt' --scores 'aescore.txt'
fi

if [ $1 == "DOS" ]; then
  echo "DOS"
  echo "Vanilla MSTREAM"
  mstream/mstream -t 'data/dostime.txt' -n 'data/dosnumeric.txt' -c 'data/doscateg.txt' -o 'score.txt' -a 0.95
  python3 results.py --label 'data/dos_label.txt' --scores 'score.txt'

  echo "MSTREAM-PCA"
  python3 pca.py --input 'data/dosnumeric.txt'
  mstream/mstream -t 'data/dostime.txt' -n 'pca.txt' -c 'data/doscateg.txt' -o 'pcascore.txt' -a 0.95
  python3 results.py --label 'data/dos_label.txt' --scores 'pcascore.txt'

  echo "MSTREAM-IB"
  python3 ib.py --input 'data/dosnumeric.txt' --inputdim 76 --label 'data/kdd_label.txt' --lr 0.01 --numEpochs 200
  mstream/mstream -t 'data/dostime.txt' -n 'ib.txt' -c 'data/doscateg.txt' -o 'ibscore.txt' -a 0.95
  python3 results.py --label 'data/dos_label.txt' --scores 'ibscore.txt'

  echo "MSTREAM-AE"
  python3 ae.py --input 'data/dosnumeric.txt' --inputdim 76 --lr 0.0001 --numEpochs 1000
  mstream/mstream -t 'data/dostime.txt' -n 'ae.txt' -c 'data/doscateg.txt' -o 'aescore.txt' -a 0.95
  python3 results.py --label 'data/dos_label.txt' --scores 'aescore.txt'
fi

if [ $1 == "DDOS" ]; then
  echo "DDOS"
  echo "Vanilla MSTREAM"
  mstream/mstream -t 'data/ddostime.txt' -n 'data/ddosnumeric.txt' -c 'data/ddoscateg.txt' -o 'score.txt' -a 0.95
  python3 results.py --label 'data/ddos_label.txt' --scores 'score.txt'

  echo "MSTREAM-PCA"
  python3 pca.py --input 'data/ddosnumeric.txt'
  mstream/mstream -t 'data/ddostime.txt' -n 'pca.txt' -c 'data/ddoscateg.txt' -o 'pcascore.txt' -a 0.95
  python3 results.py --label 'data/ddos_label.txt' --scores 'pcascore.txt'

  echo "MSTREAM-IB"
  python3 ib.py --input 'data/ddosnumeric.txt' --inputdim 76 --label 'data/ddos_label.txt' --lr 0.001 --numEpochs 200
  mstream/mstream -t 'data/ddostime.txt' -n 'ib.txt' -c 'data/ddoscateg.txt' -o 'ibscore.txt' -a 0.95
  python3 results.py --label 'data/ddos_label.txt' --scores 'ibscore.txt'

  echo "MSTREAM-AE"
  python3 ae.py --input 'data/ddosnumeric.txt' --inputdim 76 --lr 0.001 --numEpochs 100
  mstream/mstream -t 'data/ddostime.txt' -n 'ae.txt' -c 'data/ddoscateg.txt' -o 'aescore.txt' -a 0.95
  python3 results.py --label 'data/ddos_label.txt' --scores 'aescore.txt'
fi

import csv
import os

def prep_german():
    print("Preparing german_credit_data.csv...")
    in_file = 'data/german_credit_data.csv'
    out_file = 'data/german_ready.csv'
    if not os.path.exists(in_file):
        print(f"Skipping {in_file}, not found.")
        return
        
    with open(in_file, 'r', newline='', encoding='utf-8') as f_in, \
         open(out_file, 'w', newline='', encoding='utf-8') as f_out:
         
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        for row in reader:
            # map kredit: 1 (good) -> 0 (non-default), 0 (bad) -> 1 (default)
            if row['kredit'] == '1':
                row['kredit'] = '0'
            elif row['kredit'] == '0':
                row['kredit'] = '1'
            writer.writerow(row)
    print(f"Saved {out_file}")

def prep_loan():
    print("Preparing loan.csv (Lending Club)...")
    in_file = 'data/loan.csv'
    out_file = 'data/loan_ready.csv'
    if not os.path.exists(in_file):
        print(f"Skipping {in_file}, not found.")
        return
        
    with open(in_file, 'r', newline='', encoding='utf-8') as f_in, \
         open(out_file, 'w', newline='', encoding='utf-8') as f_out:
         
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        valid_statuses = {'Fully Paid', 'Charged Off', 'Default'}
        
        count = 0
        for row in reader:
            status = row['loan_status']
            if status in valid_statuses:
                # Map to binary target
                if status == 'Fully Paid':
                    row['loan_status'] = '0'
                else:
                    row['loan_status'] = '1'
                writer.writerow(row)
                count += 1
    print(f"Saved {out_file} with {count} records (filtered out 'Current' and other incomplete loans).")

if __name__ == '__main__':
    prep_german()
    prep_loan()
    print("Data prep complete!")

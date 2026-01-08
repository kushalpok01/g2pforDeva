val_data = G2PDataset("data/val.tsv")

for epoch in range(50):  # large max, stopping will end earlier
    model.train()
    train_loss = 0

    for src_seq, tgt_seq in train_data:
        src = torch.tensor(encode(src_seq, src_vocab)).unsqueeze(1).to(DEVICE)
        tgt = torch.tensor(encode(tgt_seq, tgt_vocab)).unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        out = model(src, tgt[:-1])
        loss = loss_fn(out.view(-1, out.size(-1)), tgt[1:].view(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    val_loss = evaluate(model, val_data)

    print(f"Epoch {epoch+1}")
    print(f"  Train loss: {train_loss:.3f}")
    print(f"  Val   loss: {val_loss:.3f}")

    # ---- EARLY STOPPING ----
    if val_loss < best_val_loss - 0.01:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "nepali_g2p_best.pt")
        print("  ✓ Best model saved")
    else:
        patience_counter += 1
        print(f"  ✗ No improvement ({patience_counter}/{patience})")

    if patience_counter >= patience:
        print("🛑 Early stopping triggered")
        break
